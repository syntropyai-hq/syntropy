// SPDX-License-Identifier: MIT OR Apache-2.0

//! Interface enumeration on the host that runs the test (DIST-E6-T01, RFC-0030 §7.1–§7.2).
//!
//! `this_host_lists_its_interfaces` runs everywhere, hosted CI included, where the interfaces
//! are a VM's virtual NICs. `rig_interfaces_are_classified` checks the classification a rig's
//! operator expects; it is ignored unless run on purpose:
//!
//! ```text
//! SYNTROPY_RIG_IFACES='en0=Ethernet,en1=WiFi,en2=ThunderboltIp,utun3=Overlay' \
//!     cargo test -p syntropy-net --test iface -- --ignored
//! ```

use std::collections::BTreeSet;

use syntropy_core::NetMedium;
use syntropy_core::resource::IfaceFlags;
use syntropy_net::iface::{Ordinals, interfaces};

#[test]
fn this_host_lists_its_interfaces() {
    let mut ordinals = Ordinals::new();
    let ifaces = interfaces(&mut ordinals).expect("this host's interfaces");
    assert!(!ifaces.is_empty(), "no interfaces at all");
    let ordinals_seen: BTreeSet<u16> = ifaces.iter().map(|i| i.ordinal).collect();
    assert_eq!(ordinals_seen.len(), ifaces.len(), "ordinals are unique");
    for iface in &ifaces {
        assert!(!iface.name.is_empty());
        assert!(
            !iface.flags.contains(IfaceFlags::TRUSTED_FABRIC),
            "{}",
            iface.name
        );
        assert!(
            iface
                .addrs
                .iter()
                .all(|a| a.scope_id.is_none() || a.addr.is_ipv6()),
            "{}",
            iface.name
        );
    }
    // The host reaches the network through something up and not loopback.
    let reachable = ifaces.iter().filter(|i| {
        i.medium != NetMedium::Loopback && i.flags.contains(IfaceFlags::UP) && !i.addrs.is_empty()
    });
    assert!(reachable.count() > 0, "{ifaces:#?}");
    // A second listing keeps every ordinal.
    let again = interfaces(&mut ordinals).expect("listed again");
    for iface in &again {
        if let Some(first) = ifaces.iter().find(|i| i.name == iface.name) {
            assert_eq!(first.ordinal, iface.ordinal, "{}", iface.name);
        }
    }
    #[cfg(target_os = "linux")]
    {
        let sysfs: BTreeSet<String> = std::fs::read_dir("/sys/class/net")
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        let listed: BTreeSet<String> = ifaces.iter().map(|i| i.name.clone()).collect();
        assert_eq!(listed, sysfs);
        if let Some(lo) = ifaces.iter().find(|i| i.name == "lo") {
            assert_eq!(lo.medium, NetMedium::Loopback);
        }
    }
}

/// Criterion 1 of DIST-E6-T01 on a rig: each interface the operator names has the medium they
/// name. Ignored by default, since only a rig has the hardware (RFC-0030 §7.2).
#[test]
#[ignore = "rig only: set SYNTROPY_RIG_IFACES and run with --ignored"]
fn rig_interfaces_are_classified() {
    let expected = std::env::var("SYNTROPY_RIG_IFACES").expect("SYNTROPY_RIG_IFACES is set");
    let ifaces = interfaces(&mut Ordinals::new()).expect("this host's interfaces");
    for pair in expected.split(',').filter(|p| !p.is_empty()) {
        let (name, medium) = pair.split_once('=').expect("name=Medium");
        let medium = match medium {
            "Ethernet" => NetMedium::Ethernet,
            "WiFi" => NetMedium::WiFi,
            "ThunderboltIp" => NetMedium::ThunderboltIp,
            "Overlay" => NetMedium::Overlay,
            "Loopback" => NetMedium::Loopback,
            other => panic!("unknown medium {other}"),
        };
        let found = ifaces
            .iter()
            .find(|i| i.name == name)
            .unwrap_or_else(|| panic!("no interface {name}: {ifaces:#?}"));
        assert_eq!(found.medium, medium, "{name}: {found:#?}");
    }
}
